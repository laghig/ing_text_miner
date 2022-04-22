# Environmental score predictor for food products

This application uses natural language processing and machine learning to predict the environmental footprint of footproducts based on packaging information.

## Dependencies
- MongoDB
- ...

## Data
- Eatfit database from the Auto-ID lab at ETH
- Open Food Facts (OFF) Database: https://ch-en.openfoodfacts.org/

## Installation
- clone this directory
- create a new virtual environment:
```
conda create -n myenv python=3.6
conda activate myenv
```
- install the required libraries:
```
pip install -r requirements.txt
```
- Download the OFF database from https://world.openfoodfacts.org/data