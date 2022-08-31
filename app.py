from flask import Flask, request, jsonify
import settings
from solver import predict_model


app = Flask(__name__)

@app.get("/review-solver/solve")
def solve():
    RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]

    review_sentence = request.args.get('review_sentence')

    pred = predict_model(review_sentence)
    output = {
        "review": review_sentence,
        "results": {}
      }
    for count, r in enumerate(RATING_ASPECTS):
        output["results"][r] = pred[count]

    return jsonify(output)


if __name__ == '__main__':
    app.run(host=settings.HOST, port=settings.PORT)