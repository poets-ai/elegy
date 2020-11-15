# update requirements
poetry export --without-hashes --dev -f requirements.txt > requirements.txt

#
cp README.md docs/index.md
cp CONTRIBUTING.md docs/guides/contributing.md
python scripts/update_docs.py
mkdocs build
mkdocs gh-deploy