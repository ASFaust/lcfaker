import pytest

from curvefaker.heldout import HeldOutError, check, check_path


@pytest.mark.parametrize("ident", [
    "taskset/FixedTextRNNClassification_imdb_patch32_GRU128_bs128/adam4p",
    "taskset/FixedTextRNNClassification_imdb_patch32_GRU128_bs128/nadamw",
    "synetune/lcbench/airlines", "yahpo/lcbench/3945", "synetune/pd1/cifar10_wideresnet_2048",
    "mfpbench/PD1-lm1b", "LCBENCH/adult",
])
def test_blocked(ident):
    with pytest.raises(HeldOutError):
        check(ident)


@pytest.mark.parametrize("ident", [
    "taskset/mlp_family_seed12/adam4p", "taskset/FixedMLP_mnist_128x128x128_bs128/adam8p",
    "taskset/rnn_text_classification_family_seed3/adam4p",  # same domain, other family: allowed
    "synetune/fcnet/protein_structure", "synetune/nasbench201/cifar10",
])
def test_allowed(ident):
    check(ident)


def test_paths():
    with pytest.raises(HeldOutError):
        check_path("/shared/work/private/rogue-dagger/data/pd1/foo.npz")
    with pytest.raises(HeldOutError):
        check_path("/shared/work/private/rogue-dagger/data/taskset")
    check_path("/shared/work/private/rogue-dagger/data/taskset_dev/x.npz")
