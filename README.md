# IT-Defense
Our code for paper '[The art of defense: letting networks fool the attacker](https://arxiv.org/abs/2104.02963)'

## Introduction

Some deep neural networks are invariant to some input transformations, such as Pointnet is permutation invariant to the input point cloud. In this paper, we demonstrated this property can be powerful in the defense of gradient based attacks. Specifically, we apply random input transformation which is invariant to networks we want to defend. Extensive experiments demonstrate that the proposed scheme outperforms the SOTA defense methods, and breaking the attack accuracy into nearly zero.



### Citation

if you find our work useful in your research, please consider citing:

```
@article{zhang2021itdefense,
  title={The art of defense: letting networks fool the attacker},
  author={Zhang, Jinlai and Liu, Binbin and Chen, Lyujie and Ouyang, Bo and Zhu, Jihong and Kuang, Minchi and Wang, Houqing and Meng, Yanmei},
  journal={arXiv preprint arXiv:2104.02963},
  year={2021}
}
```



## Usage

For example, your can insert our code in [IF-Defense baseline](https://github.com/Wuziyi616/IF-Defense/tree/main/baselines) to implement our IT-Defense.

```python
#attack_scripts/targeted_perturb_attack.py#L128 
# for input x.size() = Bx3xN
class Infer(nn.Module):
    def __init__(self, model):
        super(Infer, self).__init__()
        self.model = model
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x.data = x[:, :, torch.randperm(x.size()[2])].data
        x = self.model(x)
        return x
model = Infer(model)
```

Note that for BxNx3, our code should be `x.data = x[:, torch.randperm(x.size()[1]), :].data`

