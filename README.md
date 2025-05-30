Download link : https://programming.engineering/product/cs7643-deep-learning-problem-set-1/

CS7643: Deep Learning Problem Set 1
1 Gradient Descent

        (3 points) We often use iterative optimization algorithms such as Gradient Descent to find w that minimizes a loss function f(w). Recall that in gradient descent, we start with an initial

value of w (say w(1)) and iteratively take a step in the direction of the negative of the gradient of the objective function i.e.

                        w(t+1) = w(t) rf(w(t))
                        	

                        (1)

for learning rate > 0.

In this question, we will develop a slightly deeper understanding of this update rule, in par-ticular for minimizing a convex function f(w). Note: this analysis will not directly carry over to training neural networks since loss functions for training neural networks are typically not convex, but this will (a) develop intuition and (b) provide a starting point for research in non-convex optimization (which is beyond the scope of this class).

Recall the first-order Taylor approximation of f at w(t):

                    f(w) f(w(t)) + hw w(t); rf(w(t))i
                    	

                    (2)

When f is convex, this approximation forms a lower bound of f, i.e.

                    f(w) f(w(t)) + hw
                    	

                    w(t); rf(w(t))i
                    	

                    8w
                    	

                    (3)

                    |
                    		

                    {z
                    		

                    }
                    		

aﬃne lower bound to f( )

Since this approximation is a ‘simpler’ function than f( ), we could consider minimizing the approximation instead of f( ). Two immediate problems: (1) the approximation is aﬃne (thus unbounded from below) and (2) the approximation is faithful for w close to w(t). To solve both problems, we add a squared ‘2 proximity term to the approximation minimization:

                w
                		

                aﬃne
                	

                hlower bound to f( )
                										

                2
                	

                argmin f(w(t)) +
                	

                w w(t); rf(w(t))i +
                	

                2
                			

                w
                	

                w(t)
                			

                (4)
                					

                trade-oﬀ
                					
                	

                |
                			

                {z
                		

                }
                										
                				

                |{z}
                	

                |
                			

                {z
                			

                }
                	
                								
                										

                proximity term
                	

Notice that the optimization problem above is an unconstrained quadratic programming prob-lem, meaning that it can be solved in closed form (hint: gradients).

What is the solution w of the above optimization? What does that tell you about the gradient descent update rule? What is the relationship between and ?

    (3 points) Let’s prove a lemma that will initially seem devoid of the rest of the analysis but will come in handy in the next sub-question when we start combining things. Specifically, the analysis in this sub-question holds for any w?, but in the next sub-question we will use it for w? that minimizes f(w).

Consider a sequence of vectors v1; v2; :::; vT , and an update equation of the form w(t+1) = w(t) vt with w(1) = 0. Show that:

                    T
                    		

                    w(t) w?; v
                    		

                    jjw
                    	

                    ?
                    	

                    2
                    	

                    +
                    	

                    T
                    	
                    			

                    jj
                    	

                    v 2
                    	

                    Xt
                    									

                    X
                    	

                    (5)

                    h
                    		

                    ti2
                    			

                    =1
                    			

                    2
                    	

                    t=1 jj tjj

    (3 points) Now let’s start putting things together and analyze the convergence rate of gradient descent i.e. how fast it converges to w?.

First, show that for w = 1 PT w(t)

    t=1

                    		
                    			
                    	
                    			

            						
            								
            								

                            	

                                	
                                	
                                	

Let ( ) denote the standard sigmoid function. Now, for the following vector function:

    f1
    	

    (w1
    	

    ; w2) = eew1 +e2w2 + (ew1 + e2w2 )
    	

    (12)

    f2
    	

    (w1
    	

    ; w2) = w1w2 + max(w1; w2)
    	

    (13)

    (a) Draw the computation graph. Compute the value of f at w~ = (1;
    	

    1).

~

(b) At this w~, compute the Jacobian @@wf~ using numerical diﬀerentiation (using w = 0.01).

(c) At this w~, compute the Jacobian using forward mode auto-diﬀerentiation.

(d) At this w~, compute the Jacobian using backward mode auto-diﬀerentiation.

(e) Don’t you love that software exists to do this for us?

    Paper Review

The first of our paper reviews for this course comes from a much acclaimed spotlight presentation at NeurIPS 2019 on the topic ‘Weight Agnostic Neural Networks’ by Adam Gaier and David Ha from Google Brain.

The paper presents a very interesting proposition that, through a series of experiments, re-examines some fundamental notions about neural networks – in particular, the comparative importance of architectures and weights in a network’s predictive performance.

The paper can be viewed here. The authors have also written a blog post with intuitive visualizations to help understand its key concepts better.

Guidelines: Please restrict your reviews to no more than 350 words. The evaluation rubric for this section is as follows :

    (2 points) What is the main contribution of this paper? Briefly summarize its key insights, strengths and weaknesses.

    (2 points) What is your personal takeaway from this paper? This could be expressed either in terms of relating the approaches adopted in this paper to your traditional understanding of learning parameterized models, or potential future directions of research in the area which the authors haven’t addressed, or anything else that struck you as being noteworthy.

    Implement and train a network on CIFAR-10

Setup Instructions: Before attempting this question, look at setup instructions at here.

    (Upto 29 points) Now, we will learn how to implement a softmax classifier, vanilla neural networks (or Multi-Layer Perceptrons), and ConvNets. You will begin by writing the forward and backward passes for diﬀerent types of layers (including convolution and pooling), and then go on to train a shallow ConvNet on the CIFAR-10 dataset in Python. Next you will learn to use PyTorch, a popular open-source deep learning framework, and use it to replicate the experiments from before.

Follow the instructions provided here

4
