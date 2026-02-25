# Title

<!-- Change `kisnikser/m1p-template` to `intsystems/your-repository`-->
[![License](https://badgen.net/github/license/kisnikser/m1p-template?color=green)](https://github.com/kisnikser/m1p-template/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/kisnikser/m1p-template)](https://github.com/kisnikser/m1p-template/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/kisnikser/m1p-template.svg?color=0088ff)](https://github.com/kisnikser/m1p-template/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/kisnikser/m1p-template.svg?color=7f29d6)](https://github.com/kisnikser/m1p-template/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Denis Gudkov </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Daniil Dorin </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Andrii Hrabovyi </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

Modern vision models excel at recognition but fail to grasp geometric relationships like symmetry composition. This work investigates whether neural networks can internalize the algebraic structure of the dihedral group D₄ (square symmetries) rather than memorizing visual patterns. Using a Siamese encoder with autoregressive Transformer decoder, we train a model to predict whether two images are related by a D₄ transformation and identify the specific element. We demonstrate that the model learns true group properties: invariance to operation sequences (horizontal_flip → vertical_flip ≡ rotate_180), consistency with composition (g₂·g₁), canonical element representation, and correct rejection of unrelated pairs. Analysis of attention maps and embeddings reveals internal encoding of the D₄ multiplication table. Unlike VLMs that fail at such tasks, our architecture captures symbolic geometric structures.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Title},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
