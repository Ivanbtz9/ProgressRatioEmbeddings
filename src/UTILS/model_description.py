from prettytable import PrettyTable
import torch


def count_parameters(model,display=True):
    table = PrettyTable(["Modules", "Parameters", "Shape", "Type", "Grad"])
    total_params = 0
    nb_bytes = 0
    dtype2byte = {
        torch.float16:2,
        torch.float32:4,
        torch.float64:8,
        torch.int8 :1,
        torch.int32:4,
        torch.int64:8
        }

    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params, list(parameter.shape), parameter.dtype, parameter.requires_grad])
        total_params += params
        nb_bytes += params * dtype2byte[parameter.dtype]

    if display:
        print(table)
        print(f"Total Trainable Params: {total_params:,}".replace(',',' '))
        print(f"Total Giga Bytes: {(nb_bytes * 1e-9):,}".replace(',',' '))
    return total_params, nb_bytes


# if __name__ == "__main__":
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"   # choose a distill variant
#     dtype = torch.float16                                   # lighter than float32

#     # CPU load (works without big GPU; still needs RAM)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=dtype,
#         device_map="cpu",
#         low_cpu_mem_usage=True
#     ).eval()

#     _ = count_parameters(model)