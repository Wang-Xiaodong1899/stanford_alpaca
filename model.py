import sys
import os

import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import torch.nn as nn

from attention_raw import *
from utils import *

class Args(BasicArgs):
    #  necessary
    num_workers = 32
    eval_step = np.Infinity  # no evaluation
    save_step = 5
    max_train_samples = 10000

    epochs = 200
    task_name, method_name = BasicArgs.parse_config_name(__file__)
    log_dir = os.path.join(BasicArgs.root_dir, task_name, method_name)

    # custom config
    resume = False
    debug = False
    lr = 5e-5
    # find_unused_parameters = True

    # llama config
    hidden_size = 4096
    intermediate_size = 11008
    dropout = 0.1
    attention_dropout = 0.0
    n_layers = 32
    num_attention_heads = 32
    max_target_positions = 2048
    vocab_size = 32000
    multiple_of = 256
    initializer_range = 0.02
    norm_eps = 1e-5
    rms_norm_eps = 1e-6
    use_cache = False
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    max_seq_len = 512
    hidden_act = "silu"

    # model / data path
    tokenizer_hf_path = os.path.join(BasicArgs.root_dir, 'llama/tokenizer.model')
    dataset_cf = '/workspace/GODIVA/dataset/add_vtokens_coco_IM1w_bos.py'

    # vicuna 7b v1
    vicuna_7b_v1_state_dict = '/f_data/G/llama/vicuna_7b_v1.pth'

    # inference parameters
    max_image2text_len = 256
    temperature = 1
    top_p = 0.9

    # task name
    task_name = "image2text"

    # img_id
    image_id = 32000

    # inference config
    cut_length = 77

    # VQGAN config
    vae_cf = '/workspace/standford_alpaca/VQGan8192F8.py'
    vae = import_filename(vae_cf)
    VQVAE, args_vqvae = vae.Net, vae.args
    vae_path = os.path.join(BasicArgs.root_dir, 'vqg/VQGan8192F8.pth')
    vqvae_vocab_size = 8192
    compress_ratio = 8
    dim = 4096
    img_size = 128


def quick_freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        args = Args()
        self.args = args

        print("INFO begin to init vicuna")
        self.backbone = LlamaModeling(args)
        
        state_dict = torch.load(self.args.vicuna_7b_v1_state_dict, map_location="cpu")
        our_params_dict = {}
        for k in state_dict.keys():
            if k[:5] == 'model':
                our_params_dict[k[6:]] = state_dict[k]
            else:
                our_params_dict[k] = state_dict[k]
        adaptively_load_state_dict(self.backbone, our_params_dict)
        
        # NOTE: backbone should in float16
        self.backbone = self.backbone.half()
        
        self.vae = args.VQVAE(args.args_vqvae, mode='eval')
        self.vae.load_state_dict(file2data(args.vae_path, map_location='cpu'), strict=False)
        self.vae = quick_freeze(self.vae)
        
        self.num_image_tokens = self.args.vqvae_vocab_size
        self.img_w = max(1, self.args.img_size // self.args.compress_ratio)
        self.img_h = max(1, self.args.img_size // self.args.compress_ratio)
        self.vae_emb = nn.Embedding.from_pretrained(copy.deepcopy(self.vae.quantize.embed.weight),
                                                    freeze=True)
        self.vae_emb = quick_freeze(self.vae_emb)
        print("INFO init vae_proj")
        # learnable projection
        self.vae_proj = nn.Linear(self.vae_emb.embedding_dim, args.dim) #4096 -> 4096
        self.image_bos_emb = nn.Parameter(torch.randn(1, args.dim))  # bos embedding of image
        self.image_pos_emb = AxialPositionalEmbedding(args.dim,
                                                      axial_shape=(1, self.img_h, self.img_w))

        # load tokenizer
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.args.tokenizer_hf_path, add_bos_token=False, add_eos_token=False)
        self.llama_tokenizer_bos_id = self.llama_tokenizer.bos_token_id
        self.llama_tokenizer_eos_id = self.llama_tokenizer.eos_token_id
        self.llama_tokenizer_pad_id = self.llama_tokenizer.pad_token_id


        ### added embedding and lm head
        self.additional_lm_head = nn.Linear(args.hidden_size, self.args.vqvae_vocab_size, bias=False)
        self.additional_lm_head = self.additional_lm_head.half()
    
        trainable_params = sum(p.numel() for p in self.vae_proj.parameters())
        print(f"VAE Proj has {trainable_params * 1.e-6:.2f} M params.")

        # init optimizer
        # print("begin to init optimizer")
        # to_optimize_paras_vae_proj = [{'params': self.vae_proj.parameters()}]
        # to_optimize_bos_emb = [{'params': self.image_bos_emb}]
        # to_optimize_pos_emb = [{'params': self.image_pos_emb.parameters()}]
        # to_optimize_paras_llm = [{'params': self.backbone.parameters()}]
        
        # to_optimize_paras_additional_lm_head = [{'params': self.additional_lm_head.parameters()}]
        # to_optimize_paras = to_optimize_paras_llm + to_optimize_bos_emb + to_optimize_paras_vae_proj + to_optimize_pos_emb + to_optimize_paras_additional_lm_head
        # self.optimizers = [torch.optim.Adam(to_optimize_paras, lr=args.lr, eps=1e-4)]
    
    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return model_converted
    
    def get_token_embedding(self, input_ids, image_embs, input_prefix_len):
        # this image embs have been added pos embedding

        copy_inputs = input_ids.clone().detach()

        input_ids = input_ids.long()
        # print(type(input_ids), input_ids.size(), input_ids.device)
        input_embs = self.backbone.embed_tokens(input_ids).to(torch.float16)
        
        # naive implement
        for i in range(len(copy_inputs)):
            prefix_len = input_prefix_len[i]
            # print(f'input_ids: {input_ids[i]}')
            # print(f'after: {input_ids[i][prefix_len: prefix_len+256]}')
            input_embs[i, prefix_len+1: prefix_len+1+257] = image_embs[i] # TODO

        return input_embs
    
    def prepare_labels(self, labels, vae_tokens, input_prefix_len):
        # <start> + visionb_token + <end> + <eos>
        for i in range(len(labels)):
            prefix_len = input_prefix_len[i]
            # labels[i, prefix_len: prefix_len+257] = vae_tokens[i] + self.args.vocab_size # should add offset
            labels[i, prefix_len+1: prefix_len+1+256] = vae_tokens[i] + self.args.vocab_size # should add offset
            # print(f'labels: {labels[i]}')
        return labels
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        input_prefix_len: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
        ):
        device = next(self.parameters()).device

        if labels != None:
            # tokenize images
            images = images.to(device) # b c s s
            b, c, s, s = images.size()
            images = self.vae.get_codebook_indices(images).reshape(b, -1)
            # prepend bos token
            # new_bos_token = torch.LongTensor([[self.llama_tokenizer_pad_id]]*b).to(device)
            # new_images = torch.cat((new_bos_token, images), dim=-1)
            #labels need to specify with vision codes
            labels = self.prepare_labels(labels, images, input_prefix_len)
            
            image_embs = self.vae_emb(images)
            image_embs = self.vae_proj(image_embs).reshape(b, -1, self.args.dim)
            image_embs += self.image_pos_emb(image_embs)
            
            image_embs = image_embs.to(torch.float16).to(device)
            #NOTE: add the bos embedding
            image_embs = torch.cat((repeat(self.image_bos_emb, 'n d -> b n d', b=b), image_embs), dim=1)
            
            # image_embs have been added pos_embedding
            input_embs = self.get_token_embedding(input_ids, image_embs, input_prefix_len).to(device)
            
            hidden_states = self.backbone(
                inputs_embeds=input_embs, 
                attention_mask=attention_mask, 
                past_key_values=past_key_values, 
            )
            text_logits = self.backbone.lm_head(hidden_states['hidden_states'])
            additional_text_logits = self.additional_lm_head(hidden_states['hidden_states'])
            text_logits = torch.concat([text_logits, additional_text_logits], dim=-1).float()
            
            loss = None
            batch_size = labels.shape[0]

            # Shift so that tokens < n predict n
            shift_logits = text_logits[..., :, :].contiguous()
            shift_labels = labels[..., :].contiguous()
            loss_mask = loss_mask[..., :].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, self.args.vocab_size + self.args.vqvae_vocab_size), shift_labels.view(-1).long())
            loss = (loss.view(batch_size, -1) * loss_mask).mean()  # add loss mask
            
            output = (text_logits,) + tuple(hidden_states.values())[1:]
            return (loss,) + output
    
    @torch.no_grad()
    def generate(self, batch_input):
        pass


