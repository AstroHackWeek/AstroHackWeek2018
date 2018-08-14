# Machine Learning: building functions using data

*These are outline notes of what I ([Iain
Murray](https://homepages.inf.ed.ac.uk/imurray2/)) talked about. To ground
some of the discussion I discussed the jupyter
notebook `SDSS_first_ML_play.ipynb` that's in this repository.*


**Example functions people might want to write**:

* Is this email Spam?
    - Output: yes/no?  .../phish/virus/...?  Scores?
* Where is a face, as a bounding box?
    - Output: (x, y, w, h)?  Or classify yes/no for any box?
* Where is the best position to play on a Go board?
    - Output (x,y)? Or scores?  Uncertainty on scores to guide exploration?
* How many arms does this galaxy have?
* How many planets are around this star?
* Show me groups of different types of object in this image.
    - Clustering, an *unsupervised* task.

Does machine learning makes sense for these? It can get a bit meta; here are
some more functions we might try to learn from data:

* What is the best learning rate for my neural network?
* What are the best parameters for my neural network?
* What are my beliefs about parameters given data?

Functions I *wouldn't* tend to write (or expect machine learning to answer directly):

* `does_general_relativity_hold()`
* `analyse_data()`  (although people are trying...)
* ...

Work out how to make your problem look like a standard one. Start with simple
*supervised* methods: classification and regression.

**Two big ideas:**

* Turn data into vectors of "features".
* *Train* and *validate* lots of models. Then *test*.

Small print: Both of these ideas can be expanded or over-turned in some cases.
Some ML methods use similarities/distances between objects and needn't go via
vectors. The validation story might change if you really know what you're doing...


# Turning data into vectors

* One-hot encoding. E.g., for words
* Engineering of "words", e.g., image feature extractors.
* Other feature processing:
    - log or other transform
    - indicators of special values (e.g., for zero-inflated data)
    - turn angles *θ* into cos(*θ*) and sin(*θ*)
    - ordinal values
    - ...
* Pooling multiple things together to form one vector.
    - Possibly just add vectors together
    - Or use a neural net to transform vectors so that makes sense
    - Or use a more complicated aggregation like "self-attention" or an LSTM.
* Time series
    - extract features, e.g. spectral statistics.
    - extract features as for images (maybe on spectrogram "image").
    - recurrent nets, like LSTMs. "Attention mechanisms". ...
* ConvNets: `convolutional neural networks'
    - ML frameworks know how to download large pre-trained feature extractors.

Other data preparation issues:

In response to questions we also discussed that sometimes you want to upweight
important examples (some methods support weighting examples), or sub-sample
boring examples.


# Training, Validation, and Testing

* Loss functions: for input *x*, prediction *f(x)* and 'true' output *y*, loss *L(y, f(x))*
    - want to suffer small loss on average over future predictions
    - loss function could be mean square error. It's often negative
      log-probability: where there's a probabilistic model of the output in the
      machine learning method. You can often choose the loss function.
* We want to do well in future. Proxy: minimize training loss. Jargon: empirical
  loss (or risk) minimization (ERM)
* Minimizing training loss always picks most complicated model. Likely to
  over-fit. Use a separate *validation set* to guide model choices.
* Validation sets are also called *development* sets. Use during all your research.
* Before publishing a paper, compare your final model to baselines on a *test set*.
    - **Important:** It is easy to fool yourself if you look at test
      performance, or mistakes on the test set. Use the validation/development
      set for development. Yes, it is hard in practice never to "double-dip" a test
      set. I don't have magic answers for you.

Other details:

* Try to make validation and test sets representative of what you'll have to do
  in future. Hard. (The example in the notebook completely fails! Try
  downloading 500,000 examples from SDSS and you'll see...)
* Time series: pick validation and test sets in blocks after your training data
  so you can see how well you can forecast into the future. I'd not pick a time
  series prediction problem for your first ML project.
* K-fold cross validation. Makes everything harder to reason about and more
  expensive. For your first ML project try to use a dataset that's large enough
  where you don't need it!

