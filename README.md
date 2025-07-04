# Project : LSTM for portfolio return prediction and trading recommendation
Romain MRAD 

## Introduction

This machine learning module lets you manage a stock portfolio and recommends the best action to do on a daily basis. 
This module uses LSTM RNNs to predict stock adjusted close price and suggests weather to sell or buy shares.

## Table of contents

- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Workflow](#workflow)
    - [Day 0](#day-0)
    - [Day N](#day-n)
- [Source code](#source-code)
  - [Classes](#classes)
  - [Scripts](#scripts)

## Installation

Clone this repo: 

```bash
git clone https://github.com/romainmrad/UVProjet.git
```

Install all project dependencies:

| Using pip                           | Using Anaconda                                                                                                                                                                                                     |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ``pip install -r requirements.txt`` | If you don't have an environment configured <br/>```conda create --name UVProjet --file requirements.txt```<br/><br/>If you already have an environment configured<br/>```conda install --file requirements.txt``` |

## Usage

### Configuration

Use all [configuration files](config) to configure the project according to your needs.

### Workflow

#### Day 0

If this is the first time you are using this project, you need to run the following scripts:
- [Portfolio initialisation script](scripts/portfolio_initialisation.py)
- [Portfolio prediction script](scripts/portfolio_prediction.py)

#### Day N

If you already have an initialised portfolio and a prediction from day N-1:
- [Portfolio evaluation script](scripts/portfolio_evaluation.py) if prediction was Bullish
- [Portfolio modification script](scripts/portfolio_modification.py) if prediction was Bearish
- [Portfolio prediction script](scripts/portfolio_prediction.py)

## Source code

### Classes
- [Stock object](src/stock.py) ([UML Diagram](docs/class/stock.png))
- [Portfolio object](src/portfolio.py) ([UML Diagram](docs/class/portfolio.png))
- [Market object](src/market.py) ([UML Diagram](docs/class/market.png))

### Scripts
- [Portfolio initialisation script](scripts/portfolio_initialisation.py) initialises the optimal portfolio
([UML Diagram](docs/sequence/portfolio_initialisation.png))
- [Portfolio evaluation script](scripts/portfolio_evaluation.py) evaluates the current portfolio value and updates the 
evolution graph ([UML Diagram](docs/sequence/portfolio_evaluation.png))
- [Portfolio prediction script](scripts/portfolio_prediction.py) predicts the portfolio stock prices and suggests an 
investment if some stocks are Bearish ([UML Diagram](docs/sequence/portfolio_prediction.png))
- [Portfolio modification script](scripts/portfolio_modification.py) implements the suggested portfolio from day N-1 
- [Models training script](scripts/train_models.py) trains an LSTM model for each stock listed in the 
[market configuration file](config/market_config.json) ([UML Diagram](docs/sequence/train_models.png))

