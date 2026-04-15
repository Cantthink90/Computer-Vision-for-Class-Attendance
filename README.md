This is the program created by Avery Hillmann, Brian Rogerson, and James Davis. 

The purpose of this program is to evaluate and communicate how Singular Value Decomposition (SVD) affects the accuracy and speed of different CNN Architectures.
Below there are instructions on Running this project, both training and evaluation. As well as how the speedup is calculated and where to see that in the code.

--RUNNING-- 

-Training-
To generate the trained models run alexNet or leNet "Train" python scripts. These will run training. *NOTE* This will take a considerable amount of time and resources to run.
If new trained models are needed run use this to get new model weights.

-Evaluation-
To evaluate run the alexNet or leNet "Eval" python scripts to see the outputs of each architecture. 
Run the AlexNetEval main, the output will show 'top1', 'top5', 'ms/batch', 'img/sec', 'params', and 'speedup'

--Processing Time Improvement Calculation--

The calculations for speedup starts on Line 275 in alexNetEval.py amd Line 267 in leNetEval.py 
The speedup is calculated by dividing the average ms/batch of the original architecture with no svd by the average ms/batch of the 'new' weights so first the SVD and then the SVD with Fine tuning.

The output for alexNet is shown in line 276 and 279. 276 being the math and 279 being the output that you see after running.

The output for leNet is shown in lines 268 and 271. 268 being the math and 271 being the output that you see after running.
