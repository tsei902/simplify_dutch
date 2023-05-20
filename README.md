# simplify_dutch

This is the source code for my thesis on "Controllable Sentence Simplification in Dutch" 
in the Masters of AI at KU Leuven. 

# Data
The origin of the datasets in resources/datasets is: 
1) Wikilarge, available under: https://github.com/XingxingZhang/dress
The wikilarge data is limited the first 10000 rows. 

2) ASSET, available under: https://github.com/facebookresearch
Which both have been translated to Dutch. 

# Model
The Dutch T5 model t5-base-dutch from Hugging Face has been adopted and trained on the task 
of sentence simplification. 
The folder /saved model contains the final trained model on 10000 rows of data, as stated in the Thesis. 

# Sequence: 
1) TRAINING DATA needs preprocessing with preprocessor.py
2) Generation can be done with generate_on_pretrained.py with a prior adjustment of 
3) The generation parameters in model.simplify() where the decoding method needs to be chosen (Greedy decoding, Top-p & top-k, or Beam search with top-p and top-k)
4) Manual scoring of a generated text is possible with evaluate.py

# Further remarks: 
1) The folder resources/processed data contains the training set with the prepended control tokens
2) The folder resources/DUMPS contains the Word embeddings from Fares et al. (2017) have been used. The data is available under: http://vectors.nlpl.eu/repository. (ares, M., Kutuzov, A., Oepen, S., & Velldal, E. (2017). Word vectors, reuse, and replicability: Towards a community repository of large-text resources. Proceedings of the 21st Nordic Conference on Computational Linguistics, Gothenburg, Sweden.)
3) The folder resources/outputs/final_decoder_outputs contains the final generated text per decoding strategy (Greedy decoding, Top-p & top-k, or Beam search with top-p and top-k) for both the full test set and the sample dataset
4) The folder translations contains sampled text (106 and 84 rows) from the original English datasets (WIKILarge and ASSET), a machine-translated version as well as the human translated references. 

# Credits
The preprocessor.py and the utils.py contain code that has been adapted from https://github.com/KimChengSHEANG/TS_T5 (Sheang, K. C., & Saggion, H. (2021). Controllable Sentence Simplification with a Unified Text-to-Text Transfer Transformer.INLG 2021 International Conference on Natural Language Generation, Aberdeen, Scotland, UK.)
The preprocessor.py has been adapted to the usage of Dutch.

