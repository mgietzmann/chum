# Control Loops

I would like to get into instrumentation of fisheries. However there's a critical issue that I keep bumping into.
When considering some new kind of instrumentation how do you tell if it'll matter? The naive answer of course
is that any new data is meaningful data. For example if I can collect information on environmental factors and
relate that back to the fishery, surely I'll be able to reduce the error in my predictions, right? We believe
this because we know environmental factors affect fish. However there are two problems. For one thing my error may
be influenced very little by the new data in question - i.e. the new data may not be vary good at explaining
variance. This is pretty standard ML stuff. However there's another more complicated issue - what if having this
data would never change any of my decisions... What do I mean by this? Well every fishery is really a control loop.
Every couple of years folks come together and change regulations based on the new information from the past few years.
This means that there is an opportunity for people to pick up on issues without needing to necessarily predict them.
So, for example, if my environmental impact is a movement of the fishery, well you'll just be able to see that from
the catch data - no need to include environmental factors. Point is this - in order to understand the value of some
instrumentation, you need to understand how it'll affect the fishery *control loop* and that requires having modeled
the control loop itself. 

So let's give it a go. 

## Data

`rocklobster.csv` is from a fisheries management class from University of Florida. 