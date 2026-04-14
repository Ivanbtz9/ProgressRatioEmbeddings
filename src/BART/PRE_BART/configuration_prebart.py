from transformers.models.bart.configuration_bart import BartConfig
from transformers import AutoConfig

class PreBartConfig(BartConfig):
    model_type = "prebart"

AutoConfig.register("prebart", PreBartConfig)


# from transformers.models.bart.configuration_bart import BartConfig
# from transformers import AutoConfig

# class PreBartConfig(BartConfig):
#     model_type = "prebart"

#     def __init__(self, **kwargs): #, M_shannon: float = 512.0
#         """
#         Configuration for PreBART models.
#         """
#         super().__init__(**kwargs)
#         # self.M_shannon = M_shannon


# # --- Register config with AutoConfig ---
# AutoConfig.register("prebart", PreBartConfig)