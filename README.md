# Tinyscope

Teach a tiny text classifier with your own sentences and watch it learn, live, in your browser.

**Live demo:** _add your GitHub Pages link here after you deploy_

## What it does

- **Teach it two ideas.** Name two categories (starter: cozy vs. chaotic) and add example sentences.
- **See what the model sees.** Every example is a dot on a 2D map. The background color is the model's guess for each spot, so you can watch the decision boundary move as you add or remove examples.
- **Ask it anything.** Type a test sentence to get a prediction with a confidence bar.
- **See why.** Each word is shaded by how strongly it pushed the answer toward one idea.
- **Break it.** The "what if a word were missing?" list shows which single word flips the answer.


## How it works

1. **Features.** Each sentence becomes a 512-dimension vector of hashed word features plus hashed letter-trio features (so unseen words still carry some signal), then L2-normalized.
2. **Model.** Logistic regression trained with SGD in plain JavaScript. A few epochs run per animation frame so you can see learning happen.
3. **Map.** The 2D view is a PCA of your examples (power iteration, no libraries). The colored background is an exact slice of the full model through that plane: each pixel is mapped back to a 512-d point and scored.
4. **Explanations.** The model is linear, so each word's contribution is exactly the sum of its features' weights. Nothing is approximated.

No build step, no dependencies, no server. It is three static files.

## Run it locally

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Limits (and what I'd build next)

- Features are hashed words and letter trios, not a pretrained language model, so it learns surface patterns from your examples rather than meaning. That makes it fast and fully transparent, and it is also why it needs a handful of examples per idea.
- Next step: swap the feature step for sentence embeddings from a small model running in the browser (ONNX Runtime Web), keeping the same map, boundary, and explanation views.
- Other ideas: more than two classes, export/import your examples as JSON, and a guided "break the model" challenge.

## License

MIT
