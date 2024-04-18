package generate

import (
	"github.com/labstack/echo/v4"
)

func Routes(e *echo.Echo) {
	var r = e.Group("/generate")
	r.GET("", Generate)
}
