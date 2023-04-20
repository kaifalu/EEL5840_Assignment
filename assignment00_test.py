import assignment00

def test_assignment00():
	default_vals = dict(
	name = 'Kaifa Lu',
	username = 'kaifalu917',
	level = 'PhD',
	major = 'Urban and Regional Planning',
	programming_exp = 'some',
	python_exp = 'extensive',
	git_exp = 'none',
	ml_exp = 'some',
	topics = 'Classification and Regression',
	)
	out = assignment00.assignment00()
	for k,v in default_vals.items():
		assert default_vals[k] != out[k]




