import React, { useState } from 'react';

function ImageInput({ onImageSelected }) {
    const [imageFile, setImageFile] = useState(null);
  
    const handleImageChange = (event) => {
      const selectedFile = event.target.files[0];
      setImageFile(selectedFile);
      onImageSelected(selectedFile); // Pass the selected file to parent
    };
  
    return (
      <div className="image-input">
        <input type="file" accept="image/*" onChange={handleImageChange} />
        {imageFile && <p>Selected image: {imageFile.name}</p>}
      </div>
    );
  }
  
  export default ImageInput;
  