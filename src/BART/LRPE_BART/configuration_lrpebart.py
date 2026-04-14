from transformers.models.bart.configuration_bart import BartConfig
from transformers import AutoConfig

class LRPEBartConfig(BartConfig):
    model_type = "lrpebart"

AutoConfig.register("lrpebart", LRPEBartConfig)