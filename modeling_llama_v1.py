from transformers.models.llama.modeling_llama import * # type: ignore


class LlamaForCausalLM_v1(LlamaForCausalLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def show_cos_distance(
        self,
        input_ids: torch.LongTensor,
        layer_index: int,
    ) -> torch.Tensor:
        """
        Output the cosine distance between the input_hidden_states and output_hidden_states for a specific layer
        
        Args:
            layer_index (int): The layer index to use

        Returns:
            cosine_distance (torch.Tensor): The cosine distance between the input_hidden_states and output_hidden_states
        """
        assert layer_index > 0, "layer 0 does not have input_hidden_states"
        outputs = self.model(input_ids, output_hidden_states=True)
        input_hidden_states = outputs.hidden_states[layer_index - 1]
        output_hidden_states = outputs.hidden_states[layer_index]
        return F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)

    @torch.no_grad()
    def show_topk_token(
        self,
        input_ids: torch.LongTensor,
        layer_index: int,
        k: int = 10,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """
        Output the top k tokens for predicting the next token using a specific layer
        
        Args:
            tokenizer: The tokenizer to use
            model: The model to use
            input_text (str): The input text to use
            layer_index (int): The layer index to use
            k (int): The number of tokens to output

        Returns:
            values, tokens (tuple[torch.Tensor, list[str]]): A tuple containing the top k values and tokens
        """
        assert k <= self.vocab_size, "k cannot be greater than vocabulary size"
        outputs = self.model(input_ids, output_hidden_states=True)
        values, indices = torch.topk(self.lm_head(outputs.hidden_states[layer_index])[:, -1, :], k, dim=-1)
        return values, indices

    @torch.no_grad()
    def show_token_attention(
        self,
        input_ids: torch.LongTensor,
        layer_index: int,
        token_a_index: int,
        token_b_index: int,
    ):
        """
        Output the attention value between two tokens in the layer_index layer
        
        Args:
            tokenizer: The tokenizer to use
            model: The model to use
            input_text (str): The input text to use
            layer_index (int): The layer index to use
            token_a_index (int): The index of the first token
            token_b_index (int): The index of the second token

        Returns:
            attention (torch.Tensor): The attention value between the two tokens
        """
        # with `output_attentions=True`, calculation of attention falls back
        # to the original implementation instead of `torch.nn.functional.scaled_dot_product_attention`
        outputs = self.model(input_ids, output_attentions=True)
        # select the attention for the layer and average over the heads
        layer_attentions = outputs.attentions[layer_index].mean(dim=1)
        # select the attention between the two tokens
        return layer_attentions[:, token_a_index][:, token_b_index]


if __name__ == "__main__":
    from transformers import AutoTokenizer, BitsAndBytesConfig

    model_id = "/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = LlamaForCausalLM_v1.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
    model.eval()

    # test `show_topk_token` method
    input_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nWhat is the meaning of life?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    inputs = tokenizer(input_text, return_tensors="pt")
    values, indices = model.show_topk_token(inputs.input_ids, -1, 10)
    print(tokenizer.batch_decode(indices.t()))