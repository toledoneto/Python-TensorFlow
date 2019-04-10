# Autoencoder

* É uma RN muito simples, semelhante ao perceptron de multiplas camadas
* Feita para que a camada de entrada e de saída tenha o mesmo número de neurônios

### Motivação
* A ideia é reproduzir a entrada na camada de saída,
* As camadas internas (ci) são menores que as camadas de input/output
    * ci devem manter a informação e, para isso, devem fazê-lo com menos neurônios
        * esse processo permite extrair informações de um grupo de entrada
        * estamos treinando a ci para fazer uma versão comprimida dos nossos dados
   * Basicamente é uma redução de dimensões
   
### Linear Autoencoder
* Podem ser usados para fazer Principal Component Analysis (PCA), o que nos permite reduzir as dimensões dos dados
* não é uma escolha de certas dimensões e descarte de outras, é DE FATO moldar dados em dimensões menores, sem perda de informação