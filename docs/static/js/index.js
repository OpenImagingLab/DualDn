window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})

function keyHandler(event) {	
	var key_code = event.keyCode;

	element = document.getElementById('viewer');

	switch( key_code ) { 
	case 49: // 1
		replaceImage(img1,viewer);
		break;
	case 50: // 2
		replaceImage(img2,viewer);
		break;
	case 51: // 3
		replaceImage(img3,viewer);
		break;
	case 52: // 4
		replaceImage(img4,viewer);
		break;
	case 53: // 5
		replaceImage(img5,viewer);
		break;
	case 81: // Q
		replaceImage(img6,viewer2);
		break;
	case 87: // W
		replaceImage(img7,viewer2);
		break;
	case 69: // E
		replaceImage(img8,viewer2);
		break;
	case 82: // R
		replaceImage(img9,viewer2);
		break;
	case 84: // T
		replaceImage(img10,viewer2);
		break;
	}
}

function replaceImage(newimage,image)
{
	image.src=newimage.src;
	image.parentNode.href=newimage.src;
	image.style.borderColor=newimage.style.borderColor;

	// swap new image in for zoom
	var ez = $('#'+image.id).data('ezPlus');
	ez.swaptheimage(newimage.src,newimage.src); 
}


function setBorder(image)
{
	if(image.src.indexOf('_noisy.png')!=-1)
	{
		image.style.borderColor='#9E9E9E';
	}
	else if(image.src.indexOf('_maskdn.png')!=-1)
	{
		image.style.borderColor='#FFEB3B';
	}
	else if(image.src.indexOf('_restormer.png')!=-1)
	{
		image.style.borderColor='#8BC34A';
	}
	else if(image.src.indexOf('_camera.png')!=-1)
	{
		image.style.borderColor='#03A9F4';
	}
	else if(image.src.indexOf('_dualdn.png')!=-1)
	{
		image.style.borderColor='#F44336';
	}
}

function setBorder2(image)
{
	if(image.src.indexOf('_noisy.png')!=-1)
	{
		image.style.borderColor='#9E9E9E';
	}
	else if(image.src.indexOf('_maskdn.png')!=-1)
	{
		image.style.borderColor='#FFEB3B';
	}
	else if(image.src.indexOf('_cycleisp.png')!=-1)
	{
		image.style.borderColor='#8BC34A';
	}
	else if(image.src.indexOf('_restormer.png')!=-1)
	{
		image.style.borderColor='#03A9F4';
	}
	else if(image.src.indexOf('_dualdn.png')!=-1)
	{
		image.style.borderColor='#F44336';
	}
}

function Change_real_captured_results(scene) {
	// Change the main image based on the scene name
	const viewer = document.getElementById("viewer");
	const sceneTitle = document.getElementById("sceneTitle");

	// Update the title and main image source based on the scene
	sceneTitle.textContent = scene.charAt(0).toUpperCase() + scene.slice(1);
	viewer.src = `static/images/real/${get_real_captured_SceneIndex(scene)}_dualdn.png`;

	var ez = $('#viewer').data('ezPlus');
	ez.swaptheimage(viewer.src,viewer.src); 

	// Update thumbnails to show different images based on the scene
	document.getElementById("img1").src = `static/images/real/${get_real_captured_SceneIndex(scene)}_noisy.png`;
	document.getElementById("img2").src = `static/images/real/${get_real_captured_SceneIndex(scene)}_maskdn.png`;
	document.getElementById("img3").src = `static/images/real/${get_real_captured_SceneIndex(scene)}_restormer.png`;
	document.getElementById("img4").src = `static/images/real/${get_real_captured_SceneIndex(scene)}_camera.png`;
	document.getElementById("img5").src = `static/images/real/${get_real_captured_SceneIndex(scene)}_dualdn.png`;
}

function get_real_captured_SceneIndex(scene) {
	const scenes = ["duck", "plaza", "tower", "sky", "overtime", "litup", "tenement", "pagoda", "clock"];
	return scenes.indexOf(scene)+1;
}

function Change_dnd_results(scene) {
	// Change the main image based on the scene name
	const viewer = document.getElementById("viewer2");
	const sceneTitle = document.getElementById("sceneTitle2");

	// Update the title and main image source based on the scene
	sceneTitle.textContent = scene.charAt(0).toUpperCase() + scene.slice(1);
	viewer.src = `static/images/dnd/${get_dnd_SceneIndex(scene)}_dualdn.png`;

	var ez = $('#viewer2').data('ezPlus');
	ez.swaptheimage(viewer.src,viewer.src); 

	// Update thumbnails to show different images based on the scene
	document.getElementById("img6").src = `static/images/dnd/${get_dnd_SceneIndex(scene)}_noisy.png`;
	document.getElementById("img7").src = `static/images/dnd/${get_dnd_SceneIndex(scene)}_maskdn.png`;
	document.getElementById("img8").src = `static/images/dnd/${get_dnd_SceneIndex(scene)}_cycleisp.png`;
	document.getElementById("img9").src = `static/images/dnd/${get_dnd_SceneIndex(scene)}_restormer.png`;
	document.getElementById("img10").src = `static/images/dnd/${get_dnd_SceneIndex(scene)}_dualdn.png`;
}

function get_dnd_SceneIndex(scene) {
	const scenes = ["0001-18", "0014-12", "0023-17", "0026-02", "0026-04", "0035-02", "0039-13", "0049-14"];
	return scenes.indexOf(scene)+1;
}