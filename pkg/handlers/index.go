package handlers

import (
	_ "embed"

	"github.com/labstack/echo/v4"
)

//go:embed index.html
var indexHTML []byte

// func Index(c echo.Context) (err error) {
// 	return c.HTMLBlob(200, indexHTML)
// }

func Index(c echo.Context) (err error) {
	return c.File("./pkg/handlers/index.html")
}
