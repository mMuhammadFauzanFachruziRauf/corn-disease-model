const tf = require('@tensorflow/tfjs-node');
 
async function predictClassification(model, image) {
  const tensor = tf.node
    .decodeJpeg(image)
    .resizeNearestNeighbor([224, 224])
    .expandDims()
    .toFloat()
 
  const prediction = model.predict(tensor);
  const score = await prediction.data();
  const confidenceScore = Math.max(...score) * 100;

  const classes = ['Antraknose', 'Batang Jagung Sehat', 'Bercak Daun Abu-abu'];
 
  const classResult = tf.argMax(prediction, 1).dataSync()[0];
  const label = classes[classResult];
 
  let explanation, suggestion;
 
    if (label === 'Antraknose') {
    explanation = "Antraknose adalah penyakit pada jagung yang disebabkan oleh jamur Colletotrichum graminicola, yang menyebabkan bercak-bercak gelap pada daun dan batang jagung.";
    suggestion = "Lakukan rotasi tanaman dan gunakan fungisida yang direkomendasikan untuk mencegah penyebaran penyakit.";
    }

    if (label === 'Batang Jagung Sehat') {
    explanation = "Batang Jagung Sehat menunjukkan bahwa tanaman jagung dalam kondisi baik tanpa adanya gejala penyakit atau kerusakan.";
    suggestion = "Pertahankan kondisi lingkungan yang optimal dan berikan pemupukan yang cukup untuk menjaga kesehatan tanaman.";
    }

    if (label === 'Bercak Daun Abu-abu') {
    explanation = "Bercak Daun Abu-abu adalah penyakit yang disebabkan oleh jamur Cercospora zeae-maydis, yang ditandai dengan bercak abu-abu memanjang pada daun.";
    suggestion = "Gunakan varietas jagung yang tahan penyakit dan aplikasikan fungisida sesuai anjuran.";
    }

    if (label === 'Busuk Batang') {
    explanation = "Busuk Batang adalah penyakit yang disebabkan oleh jamur seperti Fusarium atau bakteri, yang menyebabkan batang jagung melemah dan akhirnya roboh.";
    suggestion = "Lakukan sanitasi lahan dengan baik dan tanam varietas yang tahan penyakit.";
    }

    if (label === 'Daun Jagung Sehat') {
    explanation = "Daun Jagung Sehat menunjukkan bahwa daun jagung dalam kondisi baik tanpa adanya tanda-tanda penyakit atau kerusakan.";
    suggestion = "Lakukan pemantauan rutin untuk memastikan tanaman tetap dalam kondisi sehat.";
    }

    if (label === 'Hawar Daun') {
    explanation = "Hawar Daun adalah penyakit yang disebabkan oleh bakteri seperti Pseudomonas atau Xanthomonas, yang menyebabkan daun jagung menguning dan layu.";
    suggestion = "Gunakan benih yang bebas penyakit dan lakukan penyemprotan bakterisida jika diperlukan.";
    }

    if (label === 'Karat Daun Jagung') {
    explanation = "Karat Daun Jagung adalah penyakit yang disebabkan oleh jamur Puccinia sorghi, yang menyebabkan bintik-bintik karat pada daun jagung.";
    suggestion = "Gunakan varietas yang tahan karat dan aplikasikan fungisida jika diperlukan.";
    }


  return { confidenceScore, label, explanation, suggestion };
}
 
module.exports = predictClassification;