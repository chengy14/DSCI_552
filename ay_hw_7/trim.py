# 
__author__ = 'Aaron Yang'
__email__ = 'byang971@usc.edu'
__date__ = '11/18/2019 9:17 PM'

if __name__ == "__main__":
	import string

	s = """The present work is intended as an investigation of certain 
problems concerning empirical knowledge. As opposed to tradi- 
tional theory of knowledge, the method adopted differs chiefly 
in the importance attached to linguistic considerations  I propose 
to consider language in relation to two main problems, which, in 
preliminary and not very precise terms, may be stated as follows : 

. What is meant by "empirical evidence for the truth of a 

proposition" 

II. What can be inferred from the fact that there sometimes is 

such evidence .^ 

Here, as usually in philosophy, the first difficulty is to see 
that the problem is difficult. If you say to a person untrained 




> """
	s = s.translate(str.maketrans('', '', string.punctuation))
	print(s)