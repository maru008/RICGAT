import argparse
import wandb
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run experiments with different parameters")
    # グラフに関するパラメータ
    parser.add_argument("-L0","--level0" , type= float, default=0.9, help="Level 0 での類似度計算")
    parser.add_argument("-L1","--level1" , type= float, default=0.75, help="Level 1 での類似度計算")
    parser.add_argument("-L2","--level2" , type= float, default=0.6, help="Level 2 での類似度計算")
    parser.add_argument("-L3","--level3" , type= float, default=0.5, help="Level 3 での類似度計算")
    parser.add_argument("-L4","--level4" , type= float, default=0.3, help="Level 4 での類似度計算")
    parser.add_argument("-L5","--level5" , type= float, default=0.1, help="Level 5 での類似度計算")
    
    parser.add_argument("-C","--compound" , type= float, default=0.75, help="Level 5 での類似度計算")
    
    parser.add_argument("--config", type=str, default="config_server.yaml", help="Config fileのパス")
    
    # 学習に関するパラメータ
    
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()


if __name__ == "__main__":
    main()