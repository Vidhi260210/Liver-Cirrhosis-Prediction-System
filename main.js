// document.getElementById('submitBtn').addEventListener('click', function() {
//     var formData = new FormData(document.getElementById('predictionForm'));
//     fetch('/predict', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => {
//         if (response.ok) {
//             return response.json();
//         } else {
//             throw new Error('Failed to submit data');
//         }
//     })
//     .then(data => {
//         document.getElementById('predictionResult').textContent = data.prediction;
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
//     console.log(formData);
// });

const submitBtn = document.querySelector('#submitBtn');

submitBtn.addEventListener('click', handleSubmit);

function handleSubmit(e) {
  e.preventDefault();

  //   if (isValid()) {
  const data = {
    drug: document.querySelector('#drug').value,
    age: document.querySelector('#age').value,
    gender: document.querySelector('#gender').value,
    ascites: document.querySelector('#ascites').value,
    hepatomegaly: document.querySelector('#hepatomegaly').value,
    spiders: document.querySelector('#spiders').value,
    edema: document.querySelector('#edema').value,
    bilirubin: document.querySelector('#bilirubin').value,
    cholesterol: document.querySelector('#cholesterol').value,
    albumin: document.querySelector('#albumin').value,
    copper: document.querySelector('#copper').value,
    alk_phos: document.querySelector('#alk_phos').value,
    sgot: document.querySelector('#sgot').value,
    tryglicerides: document.querySelector('#tryglicerides').value,
    platelets: document.querySelector('#platelets').value,
    prothrombin: document.querySelector('#prothrombin').value,
  };

  const dataArray = Object.values(data);
  console.log(dataArray);

  const jsonData = JSON.stringify(dataArray);

  const xhr = new XMLHttpRequest();
  const url = 'http://localhost:5000/';
  xhr.open('POST', url, true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.withCredentials = true;
  xhr.onreadystatechange = function () {
    if (xhr.readyState === XMLHttpRequest.DONE) {
      if (xhr.status === 200) {
        // append data to innerHTML
        const response = JSON.parse(xhr.responseText);
        document.querySelector('#predictionResult').classList.remove('hidden');
        document.querySelector('#predictionResult').innerText = response.data;
      } else {
        console.log('Error occurred');
      }
    }
  };
  xhr.send(jsonData);
  // // Send the JSON data
  //   } else {
  //     alert('Enter All Parameters');
  //   }
}

// function isValid() {
//   const requiredInputs = document.querySelectorAll('.required');

//   let response = true;

//   requiredInputs.forEach((input) => {
//     if (input.value == '' || input.value == ' ') {
//       response = false;
//     }
//   });

//   return response;
// }
