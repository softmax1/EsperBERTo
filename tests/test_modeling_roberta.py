from pytest import fixture, approx
from torch import LongTensor, randn, isfinite
from transformers import RobertaForMaskedLM

from src.modeling_roberta import (
    RobertaForMaskedLMSoftmax1,
    RobertaSelfAttentionSoftmax1,
    RobertaAttentionSoftmax1,
    RobertaLayerSoftmax1,
    RobertaEncoderSoftmax1,
    RobertaModelSoftmax1,
    RobertaConfigSoftmax1
)


@fixture(scope='session')
def num_params_in_millions() -> int:
    return 84


@fixture(scope='session')
def hidden_size() -> int:
    return 768


@fixture(scope='session')
def config() -> RobertaConfigSoftmax1:
    return RobertaConfigSoftmax1(
        vocab_size=52_032,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        n=1
    )


def test_roberta_for_masked_lm_softmax0(config, num_params_in_millions):
    model_0 = RobertaForMaskedLM(config=config)
    assert model_0.num_parameters() / 1e6 == approx(num_params_in_millions, abs=1)


def test_roberta_for_masked_lm_softmax1(config, hidden_size, num_params_in_millions):
    model_1 = RobertaForMaskedLMSoftmax1(config=config)
    assert model_1.num_parameters() / 1e6 == approx(num_params_in_millions, abs=1)

    model_0 = RobertaForMaskedLM(config=config)
    assert model_1.num_parameters() == model_0.num_parameters()

    n_samples = 3
    input_ids = LongTensor([[x] for x in range(n_samples)])
    output_1 = model_1(input_ids=input_ids)
    assert isfinite(output_1.logits).all()
    inputs_embeds = randn(n_samples, config.max_position_embeddings - 1 - n_samples, hidden_size)  # the sized are fixed here
    output_2 = model_1(inputs_embeds=inputs_embeds)
    assert isfinite(output_2.logits).all()


def test_roberta_self_attention_softmax1(config, hidden_size):
    self_attention = RobertaSelfAttentionSoftmax1(config)
    input = randn(1, hidden_size, hidden_size)  # the hidden_size must be used here
    output = self_attention(input)
    assert isfinite(output[0]).all()


def test_roberta_attention_softmax1(config, hidden_size):
    attention = RobertaAttentionSoftmax1(config)
    input = randn(2, hidden_size, hidden_size)  # the hidden_size must be used here
    output = attention(input)
    assert isfinite(output[0]).all()


def test_roberta_layer_softmax1(config, hidden_size):
    layer = RobertaLayerSoftmax1(config)
    input = randn(3, hidden_size, hidden_size)  # the hidden_size must be used here
    output = layer(input)
    assert isfinite(output[0]).all()


def test_roberts_encoder_softmax1(config, hidden_size):
    encoder = RobertaEncoderSoftmax1(config)
    input = randn(1, hidden_size, hidden_size)  # the hidden_size must be used here
    output = encoder(input)
    assert isfinite(output[0]).all()


def test_roberta_model_softmax1(config, hidden_size, num_params_in_millions):
    model = RobertaModelSoftmax1(config=config)
    n_samples = 3
    input_ids = LongTensor([[x] for x in range(n_samples)])
    output_1 = model(input_ids=input_ids)
    assert isfinite(output_1[0]).all()
    inputs_embeds = randn(n_samples, 125, hidden_size)  # the size of the middle dimension isn't fixed here
    output_2 = model(inputs_embeds=inputs_embeds)
    assert isfinite(output_2[0]).all()
