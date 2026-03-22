from src.training.BcdYoloTrainer import BcdYoloTrainer


def train_yolo():

    
    trainer = BcdYoloTrainer(name='BCD-Best')
    trainer.setup()
    trainer.run()

if __name__ == "__main__":
    train_yolo()