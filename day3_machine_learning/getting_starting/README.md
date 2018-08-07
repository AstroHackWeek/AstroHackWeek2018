# Machine Learning: building functions using data

*These are outline notes of what I ([Iain
Murray](https://homepages.inf.ed.ac.uk/imurray2/)) will talk about. When we get
to taking about processing data, I'll bring up [the jupyter
notebook](https://github.com/imurray/scratch/blob/master/SDSS_first_ML_play.ipynb).*


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
* Pooling multiple things together to form one vector
* Time series
    - extract features, e.g. spectral statistics. Extract features as for images
      (maybe on spectrum). Recurrent nets. "Attention mechanisms". ...
* ConvNets: `convolutional neural networks'


# Training, Validation, and Testing

* Loss functions: for input *x*, prediction *f(x)* and 'true' output *y*, loss *L(y, f(x))*
    - want to suffer small loss on average over future predictions
* Proxy: minimizing training loss. Empirical loss (or risk) minimization, (ERM)
* Minimizing training loss always picks most complicated model. Likely to over-fit. Use a separate *validation set* to guide model choices.
* Validation sets are also called *development* sets. Use during all your research.
* Before publishing a paper, compare your final model to baselines on a *test set*.
    - **Important:** It is easy to fool yourself if you look at test performance, or mistakes on the test set. Use the validation/development set for development.

Other details:

* Make validation and test sets representative. Hard.
* Time series: pick validation and test sets in blocks after your training data.
* K-fold cross validation. Makes everything harder to reason about and more
  expensive. For your first ML project try to use a dataset that's large enough
  where you don't need it!

