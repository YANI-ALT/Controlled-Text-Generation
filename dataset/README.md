## dataset

We use the E2E dataset in this project. The folder details are below.

Folder Structure :
```bash
dataset
├── README.md
├── control_target
│   ├── target_attribute.json
│   ├── target_compose.json
│   ├── target_compose_pos_attr.json
│   ├── target_pos.json
│   ├── target_spans.json
│   └── target_tree.json
├── control_target.zip
└── e2e_data
    ├── multicontrol.txt
    ├── multicontrol_200.txt
    ├── multicontrol_filter.txt
    ├── src1_test.txt
    ├── src1_train.txt
    ├── src1_valid.txt
    └── test_perplexity.txt

```

- ```control_target``` : This is obtained from [XiangLi1999/Diffusion-LM/tree/main/datasets/control_target](https://github.com/XiangLi1999/Diffusion-LM/tree/main/datasets/control_target)

- ```e2e_data``` : This is obtained from [XiangLi1999/Diffusion-LM/tree/main/datasets/e2e_data](https://github.com/XiangLi1999/Diffusion-LM/tree/main/datasets/e2e_data). The ```multicontrol_filter.txt``` file is obtained by listing all the possibilites for (type,area,food). The other files contain samples of controls from ```e2e_data/src1_valid.txt```.