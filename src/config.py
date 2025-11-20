import torch
class Config:
    DATA_DIR = 'data'
    RAW_DATA_DIR = 'data/raw'
    PROCESSED_DATA_DIR = 'data/processed'
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'

    MODEL_NAME = 'bert-base-uncased'
    NUM_CLASSES = 11
    MAX_LENGTH = 512

    BATCH_SIZE = 8
    NUM_EPOCHS = 1
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                         else "cuda" if torch.cuda.is_available() 
                         else "cpu")
    SEED = 42
    CATEGORY_NAMES = [
        'math.AC',  # Commutative Algebra
        'cs.CV',    # Computer Vision
        'cs.AI',    # Artificial Intelligence
        'cs.SY',    # Systems and Control
        'math.GR',  # Group Theory
        'cs.CE',    # Computational Engineering
        'cs.PL',    # Programming Languages
        'cs.IT',    # Information Theory
        'cs.DS',    # Data Structures and Algorithms
        'cs.NE',    # Neural and Evolutionary Computing
        'math.ST'   # Statistics Theory
    ]

config = Config()