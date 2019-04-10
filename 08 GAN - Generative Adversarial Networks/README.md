# GAN - Generative Adversarial Networks

* Apresentadas a primeira vez em 2014
* Têm a capacidade de criar novas amostras baseadas nas amostras usadas para treiná-lo

### Generator VS Discriminator
* Generator: gera as amostras cada vez melhores a partir do diagnóstico do Discriminator
* Discriminator:  tentar prever se são falsas

### Problemas
* Discriminator pode começar a declarar tudo como falso
    * usar uma função diferente da signóide, uma que não seja tão binária
* Mode Collapse:
    * Generator encontra uma fraqueza do Discriminator e passa a fazer amostras focadas nesse ponto
    * Podemos mudar a tx de aprendizado do Discriminator ou mesmo suas camadas, deixando-o diferente
* Treinamento podem levar dias ou semanas