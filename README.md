![D-Wave Logo](dwave_logo.png)

# Quadratic Assignment Problems

Describe your example and specify what it is demonstrating. Consider the
following questions:

* Is it pedagogical or a usable application?
* Does it belong to a particular domain such as material simulation or logistics? 
* What level of Ocean proficiency does it target (beginner, advanced, pro)? 

A clear description allows us to properly categorize your example.

Images are encouraged. If your example produces a visualization, consider
displaying it here.

Quadratic Assignment Problems, or QAP, make up a well-known class of combinatorial optimization problems which have been described as the "hardest of the hard" [Sahni and Gonzalez 1976 FIX]. They have been applied to factory and hosptial layouts as well as electronic chip design FIX CITE ALL.

## Problem Statement
Consider a manufacturing center which needs to have $n$ facilities inside of it. Each facility must be placed in one of $n$ locations. Further, each facility has some material flow between itself and other facilities. Each location must hold one facility and vice-versa. How do we place the facilities to minimize the overall flow and distance between facilities? 

QAP problems are described by zero-diagonal $n\times n$ matrices $A,B$ which represent flow and distance respectively. $A_{jk}$ represents the material flow between facilities $j,k$ while $B_{mn}$ represents the distance between locations $m,n$. The problem will also require $n^2$ binary variables $x_{jk}$ which equal $1$ if facility $j$ is in location $k$ and $0$ otherwise.

The objective is to minimize the flow times distance. Thus the *objective function* $C$ is given by
$$
C=\sum_{jkmn=1}^n f_{jk}d_{mn}x_{jm}x_{kn}.
$$

We also must add constraints that restrict solutions to having only one location per facility and vice-versa. We can write these as

$$
\sum_{k=1}^n x_{jk} = 1\text{ } \text{     for all facilities }j
$$

and

$$
\sum_{j=1}^n x_{jk} = 1\text{ } \text{     for all locations }k.
$$

At first glance there appear to be $2^{n^2}$ potential solutions to a QAP problem. However, when writing the variables $x_{jk}$ as an $n \times n$ matrix, constraints force any feasible solution to have the form of a permutation matrix. Thus there are only $n!$ feasible solutions.


## Usage

A simple command that runs your program. For example,

```bash
python <demo_name>.py
```

If your example requires user input, make sure to specify any input limitations.

## Code Overview

A general overview of how the code works.

We prefer descriptions in bite-sized bullet points:

* Here's an example bullet point

## Code Specifics

Notable parts of the code implementation.

This is the place to:

* Highlight a part of the code implementation
* Talk about unusual or potentially difficult parts of the code
* Explain a code decision
* Explain how parameters were tuned

Note: there is no need to repeat everything that is already well-documented in
the code.

## References

A. Person, "Title of Amazing Information", [short link
name](https://example.com/)

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.
