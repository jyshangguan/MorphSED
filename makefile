all:
	@cat makefile
doc:
	tox -e build_docs
ltest:
	tox -v -l
test:
	tox -e py38-test
develop:
	python setup.py develop
install:
	python setup.py install
