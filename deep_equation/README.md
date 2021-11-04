# Deep Equation Challenge

Package with the submission interface. It has a RandomModel so you can see the expected inputs and outputs. Also, it has a test to validate the students implementation.

The package is pip-installable, it is very easy to update it and implement the predictor for the student best model trained. 

## Requirements

* python3.7 ou superior

> Outros requisitos que seu modelo precisar deverão ser adicionados no arquivo `requirements.txt` com a versão correta!

## Install

Abra um terminal e, estando no mesmo diretório deste arquivo e do `setup.py`, rode:

```
pip install -e .
```

Pronto, você já tem o pacote instalado. 

## Test

To test all models:
```
pytest tests/
```

To test only the random model (example):
```
pytest tests/ -k test_random -s
```
> `-k` flag allows filtering some tests
> `-s` shows the print outputs, useful for debugging

To test only the model implemented by the student:
```
pytest tests/ -k test_student -s
```


## Note

The `model.py` and `train.py` files are not necessary for the submission, though, the student can use those files (and create other ones, if needed) to run everything inside this package. 

## Nota do Aluno
Eu consegui criar um CNN multi input, multi output que recebe a imagem de dois números e retorna os seus valores. Estou usando essas classes para realizar operações matematicas pedidas no predict, pois
não soube como fazer o treinamento da rede para ser capaz de interpretar operações basicas matematicas. Tive uma série de contratempos que impossibilitaram avançar mais no projeto e fazer melhorias referentes ao dataset.
Ela foi trainda com o minst e tem certa dificuldade para predições em fundo branco. Hoje, a solução recebe um arquivo txt com uma lista contendo caminho da imagem A, caminho da imagem B e operador. Ele trata as imagens
e depois as manda para predição. Com o resultado da predição realizo as operações matemáticas e retorno para o usuário o resultado das operações em uma lista. Caso, ocorra uma operação impossivel, como divisão por zero, 
o resultado recebe o valor ERROR.