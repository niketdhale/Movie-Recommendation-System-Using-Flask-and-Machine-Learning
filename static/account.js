
        var request = new XMLHttpRequest();

		var remove_btn = [...document.querySelectorAll('#remove_btn')];
		remove_btn.forEach(function(btn){
			btn.onclick = function(){
                var fav_movie = btn.closest('.fav-movie');
                var movie_to_remove =  btn.getAttribute('name');
                //console.log(fav_movie);
                request.open('POST','/remove_fav',true);
				request.setRequestHeader('content-type', 'application/x-www-form-urlencoded;charset=UTF-8');
				request.send("name=" + movie_to_remove);

                request.onreadystatechange = function(){
                    if(this.readyState == 4 && this.status == 200){
                        console.log(this.responseText);
                        fav_movie.remove();
                    }else{
                        //console.log('error');
                    }
                }
                
                
				
			}
		});
