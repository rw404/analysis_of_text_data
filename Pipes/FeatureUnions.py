from sklearn.pipeline import FatureUnion
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('parser', HTMLParser()),
    ('text_union', FeatureUnion(
        transformer_list = [
            ('entity_feature', Pipeline([
                ('entity_extractor', EntityExtractor()),
                ('entity_vect', CountVectorizer()),
                ]
            )),
            ('keyphrase_feature', Pipeline([
                ('keyphrase_extractor', KeyphraseExtractor()),
                ('keyphrase_vect', TfidVectorizer()),
                ])),
            ],
        transformer_weights = {
            'entity_feature': 0.6,
            'keyphrase_feature': 0.2,
            })
    ),
    ('clf', LogisticRegression()),
])
