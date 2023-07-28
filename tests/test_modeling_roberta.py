from pytest import fixture, approx
from torch import LongTensor, randn, isfinite
from transformers import RobertaConfig, RobertaForMaskedLM

from src.modeling_roberta import RobertaForMaskedLMSoftmax1, RobertaSelfAttentionSoftmax1


@fixture(scope='session')
def num_params_in_millions() -> int:
    return 84


@fixture(scope='session')
def hidden_size() -> int:
    return 768


@fixture(scope='session')
def config() -> RobertaConfig:
    return RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )


def test_roberta_for_masked_lm_softmax0(config, num_params_in_millions):
    model_0 = RobertaForMaskedLM(config=config)
    assert model_0.num_parameters() / 1e6 == approx(num_params_in_millions, 1)


def test_roberta_for_masked_lm_softmax1(config, hidden_size, num_params_in_millions):
    model_1 = RobertaForMaskedLMSoftmax1(config=config)
    assert model_1.num_parameters() / 1e6 == approx(num_params_in_millions, 1)

    n_samples = 3
    input_ids = LongTensor([[x] for x in range(n_samples)])
    output_1 = model_1(input_ids=input_ids)
    assert isfinite(output_1.logits).all()
    inputs_embeds = randn(n_samples, config.max_position_embeddings - 1 - n_samples, hidden_size)
    output_2 = model_1(inputs_embeds=inputs_embeds)
    assert isfinite(output_2.logits).all()


def test_roberta_self_attention_softmax1(config, hidden_size):
    self_attention = RobertaSelfAttentionSoftmax1(config)
    input = randn(1, hidden_size, hidden_size)
    output = self_attention(input)
    assert isfinite(output[0]).all()
