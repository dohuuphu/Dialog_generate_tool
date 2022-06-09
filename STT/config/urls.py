from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path, re_path
from django.views import defaults as default_views
from django.views.generic import TemplateView
from django.contrib.staticfiles.views import serve

def return_static(request, path, insecure=True, **kwargs):
  return serve(request, path, insecure, **kwargs)

urlpatterns = [
    #Static
    re_path(r'^static/(?P<path>.*)$', return_static, name='static'), #  add this line 
    # Django Admin, use {% url 'admin:index' %}
    path(settings.ADMIN_URL, admin.site.urls),
    # User management
    path("users/", include("ubold.users.urls", namespace="users")),
    path("accounts/", include("auther.urls")),
    path("api/", include("api.urls")),
    path("channel", include("channel.urls")),
    path("apps/", include("ubold.apps.urls", namespace="apps"),),
    path("layouts/", include("ubold.layouts.urls", namespace="layouts"),),
    path("pages/", include("ubold.pages.urls", namespace="pages"),),
    path("components/", include("ubold.components.urls", namespace="components"),),
    path("accounts/", include("ubold.accounts.urls", namespace="accounts"),),
    path("dashboard/", include("ubold.dashboard.urls", namespace="dashboard"),),
    path("", TemplateView.as_view(template_name="landing.html"), name="landing"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


if settings.DEBUG:
    # This allows the error pages to be debugged during development, just visit
    # these url in browser to see how these error pages look like.
    urlpatterns += [
        path(
            "400/",
            default_views.bad_request,
            kwargs={"exception": Exception("Bad Request!")},
        ),
        path(
            "403/",
            default_views.permission_denied,
            kwargs={"exception": Exception("Permission Denied")},
        ),
        path(
            "404/",
            default_views.page_not_found,
            kwargs={"exception": Exception("Page not Found")},
        ),
        path("500/", default_views.server_error),
    ]
    if "debug_toolbar" in settings.INSTALLED_APPS:
        import debug_toolbar

        urlpatterns = [path("__debug__/", include(debug_toolbar.urls))] + urlpatterns
