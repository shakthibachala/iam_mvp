from iam import Tensor

def test_tensor_shape():
    t = Tensor([2, 3, 4])
    assert t.shape() == [2, 3, 4]

def test_tensor_size():
    t = Tensor([2, 3, 4])
    assert t.size() == 24
