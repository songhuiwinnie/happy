$(document).ready(function(){
	$("#input-field").focus();
});


var app = new Vue({
    el: '#frame',
    data: {
        message: "",
        messages: [
            {
                is_system: true,
                is_link: false,
                message: "Dear valued customer, I am hui wen. How can I help you?"
            }
        ]
    },
    methods: {
        sendMessage: function(e){
            if(e.keyCode === 13 && !e.shiftKey && this.message){
                e.preventDefault();
                this.messages.push({"message": this.message, "is_system": false, "is_link":false});
                var message = this.message;
                this.message = "";
                axios.get("/api/query", {
                    params: {
                        message: message
                    }
                }).then((response) => {
                    this.messages.push({"message": response.data.message, "is_system": true, "is_link": response.data.is_link});
                    $(".messages").animate({ scrollTop: $(document).height() }, "fast");
                });
            }
        }
    }
});