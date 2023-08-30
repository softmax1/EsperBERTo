from src.analysis import compute_avg_and_std


def test_compute_avg_and_std():
    input_1 = [3, 4]
    output_1 = compute_avg_and_std(input_1)
    assert output_1['avg'] == 3.5
    assert output_1['std'] == 0.5

    input_2 = [5, 12]
    output_2 = compute_avg_and_std(input_2)
    assert output_2['avg'] == 8.5
    assert output_2['std'] == 3.5
