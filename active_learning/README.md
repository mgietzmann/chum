# Query Design

Models are useful specifically because they allow us to obtain hard-to-get things from easy-to-get things. However to build a model requires training data and that means collecting some amount of the hard-to-get stuff. So the question becomes, how do you minimize your costs?

One clear strategy is to search the literature to avoid models that are proven data hungry. Another clear strategy is to use transfer learning when you can. However, even once you've found that "best model" there is still the question - how do I pick the most informative data so I can use as little of it as possible? 

Two fields have concerned themselves with this question - active learning and optimal experiment design. And both of them have arrived at modifications of the same strategy. 

This strategy begins with making a pretty bold assumption - that your apriori model *is* correct. However a point of clarification is necessary here - when I say model I mean both the predictions and the uncertainty in those predictions. That uncertainty is important because it means the model itself knows that for any set of input data (features) the resulting output (target) will be variable. 

What this gives us is a "world builder". I can put features into the model and get a world (target) out. And, thanks to our uncertainty term, as I keep putting the same set of features in I'll get modified versions of that first world. 

The second component to this strategy is that you have a clear way of building a model from a specific world. This allows us, for each of the worlds we built in the last step, to build a corresponding model. 

Now each model is really a set of parameters and a methodology for using those parameters to go from features to a target. Therefore if we have a set of models we can actual compute the variance of those parameters. This variance really just represents how confident we would be in our model. 

At this point then we have a way to go from a specific set of features (a query of our world) to a predicted confidence in our model. 

Now comes the third component - we use this function of features to model confidence to search for the set of features that would give us the most confidence. This becomes our "optimal query".

Finally we would go query the real world, use the data acquired to update our model, and then we'd start the whole process over again. 

Alright so that's the strategy, but you should note something a little terrifying. This strategy is extremely intensive! For example if we used a genetic algorithm to run our optimization that took 1,000 generations to converge and evaluated 25 individuals it'd mean we'd be evaluating 25,000 feature sets. But if, to get good model confidence predictions, we needed to evaluate 25 different worlds per feature that would mean 625,000 model fits just to get one query!

Finding tricks to get around this is where all the diversity in active learning and optimal experiment design comes from. Optimal experiment design takes the strategy of trying to make everything symbol so that instead of actually having to look at all 25 worlds I can look at all possible worlds at once using mathematical abstractions. Active learning instead looks for heuristics that are easier to compute than the real parameter variance (things like feature entropy for example). But all told there are loads of different ways to try to optimize this problem which is where all the fun in the problem comes from. 

Here then, at least to my mind, is the challenge - by applying query design to a whole host of problems can one identify the smallest number of tricks required to solve the most problems in an efficient way? Because if you could do so, then you could provide a generalized software package for doing query design *regardless of the problem in question*. 

I think one could approach this by taking the following strategy:
1. Find query designs for a few specific cases
2. Find common abstractions and reimplement
3. Repeat until you're solving most of the problems without having to implement new code. 

Then as a really long term goal it would be neat to embed this software package behind an LLM chatbot (perhaps called "Optimist") so that researchers could interact with it like a contractor. This would mean that even if the researchers are not particularly technical they could still get access to really good query design for more or less free.  
