/*
FILE: this is NOT a full component — your actual ExerciseCard.jsx wasn't
in the scripts you uploaded (only backend/rag files were), so I can't
patch it directly. Drop this block into wherever the card currently
renders (or doesn't render) media, and adjust prop names to match your
existing component.

Backend now returns on every exercise object:
  gif_url:   string, "" if none    (24/2283 currently populated — see gap noted earlier)
  video_url: string, "" if none    (24/2283 — youtube.com/watch?v=... links)
  has_media: boolean               (true if EITHER gif_url or video_url is set)

Handles all three cases explicitly so nothing ever renders a broken image icon:
*/

function ExerciseMedia({ gif_url, video_url, name }) {
  if (video_url) {
    // Convert a youtube.com/watch?v=XXXX URL into an embeddable one
    const videoId = video_url.split("v=")[1]?.split("&")[0];
    if (videoId) {
      return (
        <div className="exercise-media">
          <iframe
            width="100%"
            height="200"
            src={`https://www.youtube.com/embed/${videoId}`}
            title={name}
            frameBorder="0"
            allowFullScreen
          />
        </div>
      );
    }
  }

  if (gif_url) {
    return (
      <img
        src={gif_url}
        alt={name}
        className="exercise-media"
        onError={(e) => { e.target.style.display = "none"; }}
      />
    );
  }

  // No media available — render nothing rather than a broken image icon.
  // Most exercises (2,259 / 2,283 right now) will hit this path, so this
  // matters: a missing-image placeholder on 99% of cards looks broken.
  return null;
}

// Usage inside your existing ExerciseCard:
//   <ExerciseMedia gif_url={exercise.gif_url} video_url={exercise.video_url} name={exercise.name} />
