# Pattern Recognition and Machine Learning - Practical Work

This is our solution for the 2019 practical work of the lecture "Pattern Recognition and Machine Learning" dealing with Vaizman's ExtraSensory dataset.

All sourcecode that is part of our solution can be found in "/src", alongside a requirement.txt provided by pip freeze. Our presentations and the final report are in "/docs". There is also a TensorFlow 2 implementation of Li's generalized maximal correlation embedding in "/maximal_correlation_embedding". This is an approach we discarded, but still wanted to turn in as part of our work.

Installing all packages in "src/requirements.txt" should be proficient to run our scripts. We use XGBoost with its effcient "gpu_hist" tree method, that runs on the GPU and needs CUDA 9 or 10. Luckily, GPU support should be installed automatically with "pip install xgboost". If this does not work for any reason, you have to adjust the tree method manually. Something like this should do the job:

for key, xgb_clf in clf.items():

  xgb_clf.set_params(tree_method="hist")
  
Plots of the feature importances for every individual problem can be found in the directory "/src/plots".
