Generator-Evaluator framework for feedgr based on Paddlepaddle

# Requirements
1. Paddlepaddle (>=1.0)
2. PARL

Will directly include in the repos soon.

# Dataset
Comming soon.

# Run
## Train Evaluators
BiRNN:

	cd app/demo
	sh shells/GE-eval-birnn train
Transformer:

	cd app/demo
	sh shells/GE-eval-trans train

## Train Generators and then Evaluate
DNN + SL

	cd app/demo
	sh shells/GE-sl-dnn train
	sh shells/GE-sl-dnn evaluate birnn 	# evaluate by birnn
	sh shells/GE-sl-dnn evaluate trans 	# evaluate by transformer
GRNN + SL

	cd app/demo
	sh shells/GE-sl-unirnn train
	sh shells/GE-sl-unirnn evaluate birnn 	# evaluate by birnn
	sh shells/GE-sl-unirnn evaluate trans 	# evaluate by transformer
GRNN + RL (with rewards from birnn)

	cd app/demo
	sh shells/GE-rl-unirnn-candenc_none-eval_birnn-0 train
	sh shells/GE-rl-unirnn-candenc_none-eval_birnn-0 evaluate 	# evaluate by birnn
GRNN + RL (with rewards from transformer)

	cd app/demo
	sh shells/GE-rl-unirnn-candenc_none-eval_trans-0 train
	sh shells/GE-rl-unirnn-candenc_none-eval_trans-0 evaluate 	# evaluate by transformer
PointerNet + RL (with rewards from birnn)

	cd app/demo
	sh shells/GE-rl-pointernet-candenc_sum-eval_birnn-0 train
	sh shells/GE-rl-pointernet-candenc_sum-eval_birnn-0 evaluate 	# evaluate by birnn
PointerNet + RL (with rewards from transformer)

	cd app/demo
	sh shells/GE-rl-pointernet-candenc_sum-eval_trans-0 train
	sh shells/GE-rl-pointernet-candenc_sum-eval_trans-0 evaluate 	# evaluate by transformer
PointerNet + RL (with rewards from log data)

	cd app/demo
	sh shells/GE-rl-pointernet-candenc_sum-eval_birnn-log_reward train
	sh shells/GE-rl-pointernet-candenc_sum-eval_birnn-log_reward evaluate 	# evaluate by birnn


