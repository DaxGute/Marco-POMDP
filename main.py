import os
import time

from MarcoPolo import MarcoPolo

POOLS_DIR = "pools"


def main():
    pool_files = [f for f in os.listdir(POOLS_DIR)]

    print("Available pool shapes:\n")
    for i, filename in enumerate(pool_files):
        print("[" + str(i) + "]" + str(filename))

    choice = input("\nEnter the number of the pool you want to load: ").strip()

    game = MarcoPolo(pool_files[int(choice)], num_polos=3)

    game.render()
    i = 1
    while not game.iterate_round():
        game.render()
        print(f"Round {i}")
        i += 1
        time.sleep(5)
    game.render()

    print("Game over!")
    print("Marco was caught by a polo!")


if __name__ == "__main__":
    main()

