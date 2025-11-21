Requirement:
• Python ≥ 3.5
• Numpy == 1.17.3 
• Keras == 2.2.4

You can run the code on whatever operating system that contains above python package.


How to run the code:
1. You can train the agents using default setting by running:
			python main.py
2. If you want to make an agent play with an agent, you can run:
			python agents_play.py --filename1 (path to the first agent) --filename2 (path to the second agent)
3. If you want to play with an agent, you can run:
			python human_play.py --filename (path to the agent)

There might be some warning messages, and it is because of the update of the tensorflow dependency. You can ignore it.