//var model = document.getElementById('mymodel');
//	var btn = document.getElementById('mybtn');
//	var span = document.getElementsByClassName("close")[0];

//	btn.onclick = function(){
//		model.style.display = "block";
//	}
//	span.onclick = function(){
//		model.style.display = "none";
//	}
//	window.onclick = function(event){
//		if(event.target == model){
//			model.style.display = "none";
//		}
//	}
/////////////////////////////////////////////////////////////////
//	function open_model(){
//		var span = document.getElementsByClassName("close")[0];
//		model.style.display = "block";
//		span.onclick = function(){
//		model.style.display = "none";
//	}
//	window.onclick = function(event){
//		if(event.target == model){
//			model.style.display = "none";
//			}
//		}
//	}
///////////////////////////////////////////////////////////////
    var modelBtns = [...document.querySelectorAll('#movie_title')];
		modelBtns.forEach(function(btn){
			btn.onclick = function(){
				var model = btn.getAttribute('name');
				// console.log(model);
				document.getElementById(model).style.display = "block";
				}
		});

		var closeBtns = [...document.querySelectorAll(".close")];
		closeBtns.forEach(function(btn){
			btn.onclick = function(){
				var model = btn.closest('.model');
				model.style.display = "none";
			}
		});

		window.onclick = function(event){
			if(event.target.className === "model"){
				event.target.style.display = "none";
			}
		}

		var searched_movie_title = document.getElementById('title');
		var searched_movie_model = document.getElementById('searched_movie_model');
		searched_movie_title.onclick = function(){
			searched_movie_model.style.display = "block";
		}

/////// Favorite Button Js ////////////////////
		var request = new XMLHttpRequest();
		var favoritebutton = [...document.querySelectorAll('#fav-btn')];

		
			// var user_type = 'guest';
			// request.open('GET','/favorite',true);
			// request.setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
			// request.send("user_type=");

			// request.onreadystatechange = function(){
			// 	if(this.readyState == 4 && this.status == 200){
			// 		console.log(this.responseText);
			// 		if(this.responseText === 'Guest'){
			// 			favoritebutton.forEach(function(button){
			// 				button.style.display = 'none';
			// 			})
			// 		}
			// 	}
			// }
		
		// var guest_user = {{ guest_user|safe }};
		// console.log(guest_user);
		
		var fav = [...document.querySelectorAll('#fav-btn')];
			fav.forEach(function(btn){
				btn.onclick = function(){
					var fav_movie_name = btn.getAttribute('name');
					//console.log(fav_movie_name);
					request.open('POST','/favorite',true);
					request.setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
					request.send("name=" + fav_movie_name);
			
					request.onreadystatechange = function(){
						if(this.readyState == 4 && this.status == 200){
							//console.log(this.responseText);
							if(this.responseText == 'Guest'){
								// alert('Login to add movies to favorites');
								swal({
									title:"Alert",
									text:"Please Login to Continue",
									icon:"warning"
								});
							}else{
								//console.log('inside buttion function');
							// result_div.innerHTML = this.responseText;
							//fav_logo.classList.replace('fa-plus','fa-check');
							var fav_logo = document.createElement('i');
							fav_logo.setAttribute('class','fa fa-check');
							fav_logo.setAttribute('aria-hidden',true);
							btn.innerHTML = 'Added To Favorites ';
							btn.appendChild(fav_logo);
							btn.disabled = true;
							// {
							// 	console.log(this.responseText);
							// 	var link_to_login = document.createElement('a');
							// 	link_to_login.setAttribute("href","{{ url_for('login') }}");
							// 	btn.appendChild(link_to_login);
							// }
							}
							
						} else{
							// result_div.innerHTML = 'error';		
						}
					}
				}
			});


///////  Account Remove Button Js ////////////
		
		var fav_movie = document.querySelectorAll('.fav-movie');
		var remove_btn = [...document.querySelectorAll('#remove_btn')];
		remove_btn.forEach(function(btn){
			btn.onclick = function(){
				var fav_movie = btn.closest('.fav-movie');
				//console.log(fav_movie);
				fav_movie.remove();
			}
		});





// function do_ajax(){
// 	var req = new XMLHttpRequest();
// 	var fav_btn = document.getElementById('fav-btn');
// 	var fav_movie_name = fav_btn.getAttribute('name');
// 	var fav_logo = document.getElementById('fav-logo');
// 	req.onreadystatechange = function(){
// 		if(this.readyState == 4 && this.status == 200){
// 			console.log(this.responseText);
// 			// result_div.innerHTML = this.responseText;
// 			fav_logo.classList.replace('fa-plus','fa-check');
// 		} else{
// 			// result_div.innerHTML = 'error';
			
// 		}
// 	}

// 	req.open('POST','/favorite',true);
// 	req.setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
// 	req.send("name=" + fav_movie_name);
// }