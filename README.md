# Syntactically Informed Text Compression with Recurrent Neural Networks

## Installation

### Using our AMI:

Start a g2.2xlarge instance using our AMI (ami-2c3a7a4c).

```
git clone https://github.com/davidcox143/rnn-text-compress.git
cd rnn-text-compress
pip install -r requirements.txt
make
```

## Usage

### Train a model
```
python rnn-text-compress.py -t < path to text file >
```

### Evaluate accuracy of a single model
```
python rnn-text-compress.py -e < path to text file > < path to weights file >
```

### Evaluate the accuracy of a model over time and plot results
```
python rnn-text-compress.py -p < path to text file > < path to weights directory >
```
