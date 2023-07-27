from pytest import fixture, approx
from transformers import RobertaConfig, RobertaForMaskedLM

from src.modeling_roberta import RobertaForMaskedLMSoftmax1


@fixture(scope='session')
def config():
    return RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )


def test_roberta_for_masked_lm_softmax0(config):
    model_0 = RobertaForMaskedLM(config=config)
    assert model_0.num_parameters() / 1e6 == approx(84, 1)


def test_roberta_for_masked_lm_softmax1(config):
    model_1 = RobertaForMaskedLMSoftmax1(config=config)
    assert model_1.num_parameters() / 1e6 == approx(84, 1)
