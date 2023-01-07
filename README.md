# Buzz
Reinforcement learning approaches to the Lunar Lander OpenAI Gym environment.

## Pre-requisites
- [Python 3.8](https://www.python.org/downloads/)
- [Swig](http://www.swig.org/download.html)

## Installation
1. Clone the repository
```bash
git clone https://github.com/CM30359-Group-2/buzz.git
```
2. Install the dependencies
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python main.py
```
Output will be saved to the same directory as the script.

## Options
| Option | Description | Default | Required |
| --- | --- | --- | --- |
| `-a` or `--agent` | The agent to use. Options are `dqn`, `ddqn`, `dqfd` (requires pre-generated demos) and `q-learning`. | `dqn` | No |
| `-e` or `--episodes` | The number of episodes to run for. | `1000` | No |
| `-s` or `--save-progress` | Whether to save the agent's progress at regular intervals | `False` | No |
| `-d` or `--demos` | The path to the demonstrations directory. Only required for `dqfd` agent. | `None` | No |
| `-c` or `--checkpoint` | The path to the checkpoint to load during evaluation. | `None` | No |
| `-r` or `--render` | Whether to render the environment. | `False` | No |

## Plotting Results
Call the `plot.py` script from the `scripts` directory with the path to the results file as an argument:
```bash
python plot.py dqfd2023-01-07_01-00-42.txt
```
This will generate a plot of the results.

## Generating Demonstrations
Call the `demos.py` script from the `scripts` directory. This will generate a folder in the root directory called `demos` containing the demonstrations. It takes two arguments:
1. The number of episodes to run (Note: this may be less than the number of demonstrations generated as only episodes where the environment is solved are saved) 
2. The path to the checkpoint to load for the Q-learning agent
```bash
python demos.py 1000 checkpoints/dict.pickle
```

## Authors
- [Max Wood](https://maxwood.tech)
- [Rowan Bevan](https://github.com/RowanBevan)
- [Cameron Grant](https://github.com/cg-2611)
- [Cameron Archibold](https://github.com/cam-arch)

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements
- [OpenAI Gym](https://gym.openai.com/)